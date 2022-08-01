#include "VectorSearch.h"
#include <cstdio>
//#include "include/faiss/index_io.h"
//#include "include/faiss/Index.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <regex>
#include "Utils.h"
//#include "include/faiss/IndexFlat.h"

//#include <direct.h>//---getcwd
//using namespace faiss;


void VectorSearch::LoadIndexFile()
{
    std::string file_path = this->index_dir + OS_PATH_SEP + "vector.index";
    const char *fname = file_path.c_str();
    if (access(fname, F_OK) != -1)
    {
        LOGD("Find!");
    }
    else
    {
        LOGD("Not Find");
    }
    this->index = faiss::read_index(fname, 0);
}


// load id_map.txt
void VectorSearch::LoadIdMap()
{
    std::string file_path = this->index_dir + OS_PATH_SEP + "id_map.txt";
    std::ifstream in(file_path);
    std::string line;
    std::vector<std::string> m_vec;
    if (in)
    {
        while (getline(in, line))
        {
            std::regex ws_re("\\s+");
            std::vector<std::string> v(
                    std::sregex_token_iterator(line.begin(), line.end(), ws_re, -1),
                    std::sregex_token_iterator()
            );
            if (v.size() != 2)
            {
                std::cout << "The number of element for each line in : " << file_path
                          << "must be 2, exit the program..." << std::endl;
                exit(1);
            }
            else
                this->id_map.insert(std::pair<long int, std::string>(std::stol(v[0], nullptr, 10), v[1]));
        }
    }
}

// doing search
const SearchResult &VectorSearch::Search(float *feature, int query_number)
{
    this->D.resize(this->return_k * query_number);
    this->I.resize(this->return_k * query_number);
    this->index->search(query_number, feature, return_k, D.data(), I.data());
    this->sr.return_k = this->return_k;
    this->sr.D = this->D;
    this->sr.I = this->I;
    return this->sr;
}

const std::string &VectorSearch::GetLabel(faiss::Index::idx_t ind)
{
    return this->id_map.at(ind);
}