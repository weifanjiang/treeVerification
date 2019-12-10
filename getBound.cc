#include <iostream>
#include <fstream>
#include <iomanip>
#include <string> 
#include <tuple>
#include <limits>
#include <algorithm>
#include <vector>
#include <math.h>
#include <chrono>
#include <random>
#include <cmath>
#include "svmreader.hpp"
#include "tree_func.hpp"

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

double compute_r_inf(interval_map<int,Interval> box) {
  double max_dist = 0.0;
  for (interval_map<int, Interval>::const_iterator it = box.cbegin(); it != box.cend(); ++it) {
    double lo = it->second.lower;
    double hi = it->second.upper;
    max_dist = max(max_dist, lo);
    max_dist = max(max_dist, hi);
  }
  return max_dist;
}

double compute_r_1(interval_map<int,Interval> box) {
  double dist = 0.0;
  for (interval_map<int, Interval>::const_iterator it = box.cbegin(); it != box.cend(); ++it) {
    double lo = it->second.lower;
    double hi = it->second.upper;
    dist = dist + max(lo, hi);
  }
  return dist;
}

int main(int argc, char** argv){

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  string config_file = string(argv[1]);
  ifstream config(config_file);
  json param;
  config >> param;
  
  string ori_file;
  string tree_file;
  int start_idx;
  int num_attack;
  int max_clique;
  int max_search;
  int max_level;
  int num_classes;
  bool dp;
  bool one_attr;
  int only_attr;
  int feature_start;
  string bound_file;

  if (param.find("bound") != param.end()){
    bound_file = param["bound"];
  }

  if (param.find("inputs") != param.end()){
    ori_file = param["inputs"];
  }
  else {
    throw invalid_argument("inputs datapoints in LIBSVM format is missing");
  }

  if (param.find("model") != param.end()){
    tree_file = param["model"];
  }
  else {
    throw invalid_argument("model is missing in config file");
  }

  if (param.find("start_idx") != param.end()){
    start_idx = int(param["start_idx"]);
  }
  else {
    throw invalid_argument("start_idx is missing in config file");
  }

  if (param.find("num_attack") != param.end()){
    num_attack = int(param["num_attack"]);
  }
  else {
    throw invalid_argument("num_attack is missing in config file");
  }

  if (param.find("max_clique") != param.end()){
    max_clique = int(param["max_clique"]);
  }
  else {
    throw invalid_argument("max_clique is missing in config file");
  }

  if (param.find("max_search") != param.end()){
    max_search = int(param["max_search"]);
  }
  else {
    throw invalid_argument("max_search is missing in config file");
  }

  if (param.find("max_level") != param.end()){
    max_level = int(param["max_level"]);
  }
  else {
    throw invalid_argument("max_level is missing in config file");
  }

  if (param.find("num_classes") != param.end()){
    num_classes = int(param["num_classes"]);
  }
  else {
    throw invalid_argument("num_classes is missing in config file");
  }

  if (param.find("dp") != param.end()){
    dp = bool(int(param["num_classes"]));
  }
  else {
    dp = false;
  }
  
  if (param.find("one_attr") != param.end()){
    one_attr = true;
    only_attr = int(param["one_attr"]);
  }
  else {
    one_attr = false;
    only_attr = -100;
  }
 
  if (param.find("feature_start") != param.end()){
    feature_start = int(param["feature_start"]);
  }
  else {
    feature_start = 1;
  }

  if (num_classes < 2) { num_classes = 2; }

  ifstream bound_data(bound_file);
  json bound_values;
  bound_data >> bound_values;

  // Construct initial box B
  interval_map<int,Interval> feature_bound;
  Interval bound_for_x;
  for (auto& element : bound_values.items()) {
    bound_for_x = {element.value()[0], element.value()[1]};
    feature_bound[stoi(element.key())] = bound_for_x;
  }
  
  // read data inputs 
  vector<vector<double>> ori_X;
  vector<int> ori_y;  
  read_libsvm(ori_file, ori_X, ori_y, num_classes<=2);
  
  ifstream tree_data(tree_file);
  json model;
  tree_data >> model;  
   
  vector<vector<Leaf>> all_tree_leaves;
  vector<Leaf> one_tree_leaves; 
  
  //calculate and print leave bounds
  for (int i=0; i<model.size(); i++){
    interval_map<int,Interval> no_constr;
    no_constr.clear();
    one_tree_leaves.clear();
    int class_label;
    if (num_classes==2)
      class_label = -1;
    else
      class_label = i % num_classes;
    dfs(model[i], i, no_constr, one_tree_leaves, class_label);
    all_tree_leaves.push_back(one_tree_leaves);
    
  } 

  high_resolution_clock::time_point t5 = high_resolution_clock::now(); 
  double avg_bound_inf = 0;
  double avg_bound_1 = 0;
  num_attack = min(int(ori_X.size())-start_idx, num_attack);
  int n_initial_success = 0;
  double best_rob_eps_inf = 1.0;
  double best_rob_eps_1 = 1.0;
  int succ_attack_count = 0;
  for (int n=start_idx; n<num_attack+start_idx; n++){ //loop all points
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    vector<bool> rob_log;
    vector<interval_map<int,Interval>> eps_log;
    int last_rob = -1;
    int last_unrob = -1;
    double rob_eps_inf = 0.0;
    double rob_eps_1 = 0.0;
    for (int search_step=0; search_step<max_search; search_step++){
      bool robust = true;
      if (num_classes <= 2){ 
        vector<double> sum_best = find_multi_level_best_score(ori_X[n], ori_y[n], -1, all_tree_leaves, num_classes, max_level, max_clique, feature_start, one_attr, only_attr, dp, feature_bound); 
        if (sum_best.size() == 0) {
          robust = false;
        } else {
          robust = (ori_y[n]<0.5&&sum_best.back()<0)||(ori_y[n]>0.5&&sum_best.back()>0);
        }
      }
      else{
        for (int neg_label=0; neg_label<num_classes; neg_label++){
          if (neg_label != ori_y[n]){
            vector<double> sum_best = find_multi_level_best_score(ori_X[n], ori_y[n], neg_label, all_tree_leaves, num_classes, max_level, max_clique, feature_start, one_attr, only_attr, dp, feature_bound);
            if (sum_best.size() == 0) {
              robust = false;
            } else {
              robust = robust && (sum_best.back()>0);
            }
          }
        }
      }
      // at the first search, evaluate the verified error 
      if (search_step == 0 && robust) {
        n_initial_success += 1;
      }
      rob_log.push_back(robust);
      eps_log.push_back(feature_bound);
      if (robust) {
        last_rob = rob_log.size() - 1;
        rob_eps_inf = max(rob_eps_inf, compute_r_inf(feature_bound));
        rob_eps_1 = max(rob_eps_1, compute_r_1(feature_bound));
      }
      else {
        last_unrob = rob_log.size() - 1;
      }
      float current_bound = compute_r_inf(feature_bound);
      if (last_rob<0) {
        interval_map<int,Interval> feature_bound_new;
        for (interval_map<int, Interval>::const_iterator it = feature_bound.cbegin(); it != feature_bound.cend(); ++it) {
          double lower = it->second.lower;
          double upper = it->second.upper;
          lower = lower * 0.5;
          upper = upper * 0.5;
          Interval current_feature_bound = {lower, upper};
          feature_bound_new[it->first] = current_feature_bound;
        }
        feature_bound = feature_bound_new;
      }
      else {
        if (last_unrob<0){ 
          if (current_bound >= 1){
            break;
          }
          interval_map<int,Interval> feature_bound_new;
          for (interval_map<int, Interval>::const_iterator it = feature_bound.cbegin(); it != feature_bound.cend(); ++it) {
            double lower = it->second.lower;
            double upper = it->second.upper;
            lower = min(1.0, 2.0 * lower);
            upper = min(1.0, 2.0 * upper);
            Interval current_feature_bound = {lower, upper};
            feature_bound_new[it->first] = current_feature_bound;
          }
          feature_bound = feature_bound_new;
        } else {
          interval_map<int,Interval> feature_bound_new;
          for (interval_map<int, Interval>::const_iterator it = feature_bound.cbegin(); it != feature_bound.cend(); ++it) {
            int attr = it->first;
            double lower = it->second.lower;
            double upper = it->second.upper;
            lower = 0.5 * (eps_log[last_rob][attr].lower + eps_log[last_unrob][attr].lower);
            upper = 0.5 * (eps_log[last_rob][attr].upper + eps_log[last_unrob][attr].upper);
            Interval current_feature_bound = {lower, upper};
            feature_bound_new[it->first] = current_feature_bound;
          }
          feature_bound = feature_bound_new;
        }
      }
    }
    
    double clique_bound_inf = 0;
    double clique_bound_1 = 0;
    if (last_rob>=0){
      clique_bound_inf = compute_r_inf(eps_log[last_rob]);
      clique_bound_1 = compute_r_1(eps_log[last_rob]);
      avg_bound_inf = avg_bound_inf + clique_bound_inf;
      avg_bound_1 = avg_bound_1 + clique_bound_1;
      succ_attack_count += 1;
      best_rob_eps_inf = min(best_rob_eps_inf, rob_eps_inf);
      best_rob_eps_1 = min(best_rob_eps_1, rob_eps_1);
      cout<< "\npoint "<< n << ": robust epsilon (linf) is " << clique_bound_inf << " " << endl;
      cout<< "\npoint "<< n << ": robust epsilon (l1) is " << clique_bound_1 << " " << endl;
    }
    else{
      cout<< "\npoint "<< n << ": WARNING! no robust eps found, verification bound is set as 0 !!!!!!!!\n";
    }
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    auto point_duration = duration_cast<microseconds>( t4 - t3).count();
  }
  double verified_err = 1.0 - n_initial_success / (double)num_attack;
  avg_bound_inf = avg_bound_inf / succ_attack_count; 
  avg_bound_1 = avg_bound_1 / succ_attack_count;
  cout << "\nclique method average linf bound:" << avg_bound_inf << endl;
  cout << "\nclique method average l1 bound:" << avg_bound_1 << endl;
  cout << "\nclique method best linf bound:" << best_rob_eps_inf << endl;
  cout << "\nclique method best l1 bound:" << best_rob_eps_1 << endl;
  cout << "verified error at initial box = " << verified_err << endl;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto total_duration = duration_cast<microseconds>( t2 - t1 ).count();
  cout << " total running time: " << double(total_duration)/1000000.0 << " seconds\n";
  cout << " per point running time: " << double(total_duration)/1000000.0/num_attack << " seconds\n";
  return 0;
}










