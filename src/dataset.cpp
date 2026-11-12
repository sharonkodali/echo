#include "echo/dataset.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace echo {

namespace {

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, '\t')) out.push_back(field);
    // getline drops a trailing empty field ("a\tb\t"); keep it so column
    // counts stay correct.
    if (!line.empty() && line.back() == '\t') out.emplace_back();
    return out;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

float parse_float(const std::string& s, std::size_t line_no) {
    // std::stof silently accepts trailing junk ("1.5abc"), so check that
    // strtof consumed the whole field. (std::from_chars would be cleaner,
    // but Apple's libc++ only recently added float support.)
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE) {
        throw std::runtime_error("line " + std::to_string(line_no) + ": not a number: '" + s + "'");
    }
    return value;
}

}  // namespace

std::uint32_t Dataset::add(std::span<const float> v, std::vector<std::string> meta) {
    if (v.size() != dim_) {
        throw std::invalid_argument("vector has " + std::to_string(v.size()) +
                                    " values, dataset expects " + std::to_string(dim_));
    }
    values_.insert(values_.end(), v.begin(), v.end());
    meta_.push_back(std::move(meta));
    return static_cast<std::uint32_t>(size() - 1);
}

std::vector<std::uint32_t> Dataset::find(std::size_t meta_col, const std::string& needle) const {
    const std::string want = lower(needle);
    std::vector<std::uint32_t> hits;
    for (std::size_t i = 0; i < meta_.size(); ++i) {
        if (meta_col < meta_[i].size() && lower(meta_[i][meta_col]).find(want) != std::string::npos) {
            hits.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return hits;
}

Dataset Dataset::load_tsv(const std::string& path, std::size_t num_meta_cols) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);

    std::string line;
    if (!std::getline(file, line)) throw std::runtime_error(path + " is empty");
    const auto header = split_tabs(line);
    if (header.size() <= num_meta_cols) {
        throw std::runtime_error(path + ": header has no feature columns");
    }

    Dataset ds(header.size() - num_meta_cols);
    const auto split = static_cast<std::ptrdiff_t>(num_meta_cols);
    ds.meta_columns_.assign(header.begin(), header.begin() + split);
    ds.feature_columns_.assign(header.begin() + split, header.end());

    std::vector<float> values(ds.dim());
    std::size_t line_no = 1;
    while (std::getline(file, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();  // Windows line endings
        if (line.empty()) continue;
        const auto fields = split_tabs(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error(path + " line " + std::to_string(line_no) + ": expected " +
                                     std::to_string(header.size()) + " columns, got " +
                                     std::to_string(fields.size()));
        }
        for (std::size_t j = 0; j < ds.dim(); ++j) {
            values[j] = parse_float(fields[num_meta_cols + j], line_no);
        }
        ds.add(values, {fields.begin(), fields.begin() + split});
    }
    return ds;
}

}  // namespace echo
