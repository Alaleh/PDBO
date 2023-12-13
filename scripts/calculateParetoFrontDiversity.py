import os
import csv
import copy
import numpy as np


def read_csv(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        csv_l = []
        for row in reader:
            csv_l.append(row)
    file.close()
    return csv_l


def write_to_file(path, v):
    with open(path, "a") as filehandle:
        for h in v:
            filehandle.write(str(h) + "\n")
    filehandle.close()


def map_to_01(cur_p, lim):
    return [(cur_p[i] - lim[i][0]) / (lim[i][1] - lim[i][0]) for i in range(len(lim))]


def euclidean(v1, v2):
    return sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]) ** .5


def calculate_pair_diversity_by_batch(outputs, batch_size, init_count=5):
    euc_sum = 0
    div_list = [0.0]

    for p in range(1, len(outputs)):
        for q in range(p):
            euc_sum += euclidean(outputs[p], outputs[q])
        div_list.append(euc_sum / (p * (p + 1) / 2.0))

    for p in range(init_count):
        div_list[p] = div_list[init_count-1]

    for p in range(init_count, len(div_list)):
        if (p - init_count + 1) % batch_size:
            div_list[p] = div_list[min((init_count-1) + (((p - (init_count-1)) // batch_size) + 1) * batch_size, len(div_list) - 1)]

    return div_list


def calculate_diversity_by_seed(all_seed_outputs, batch_size, normalize=True, limits=[]):
    seed_divs = []

    if normalize:
        for i in range(len(all_seed_outputs)):
            for j in range(len(all_seed_outputs[i])):
                all_seed_outputs[i][j] = map_to_01(all_seed_outputs[i][j], limits)

    for seed in all_seed_outputs:
        s_d = calculate_pair_diversity_by_batch(seed, batch_size)
        seed_divs.append(s_d)

    return seed_divs


def calculate_PFD():
    all_bench = ['zdt1', 'zdt2', 'zdt3', 'gtd', 'dtlz1', 'dtlz3', 'dtlz5']
    prob_details = {'zdt1': [25, 2], 'zdt2': [4, 2], 'zdt3': [12, 2], 'gtd': [4, 3],
                    'dtlz1': [10, 4], 'dtlz3': [9, 4], 'dtlz5': [12, 6]}
    all_algs = ['PDBO']
    benchmarks = os.listdir("./result")
    benchmarks = [p for p in benchmarks if p in all_bench]

    for bench in benchmarks:
        batches = os.listdir("./result/" + bench)
        batches = [b for b in batches if b.endswith('batch')]
        batches = sorted(batches, key=lambda s: int(s.split('-')[0]))
        n_var, n_obj = int(prob_details[bench][0]), int(prob_details[bench][1])
        ALL_LIM = []
        for cur_batch in batches:
            batch_size = int(copy.deepcopy(cur_batch).split('-')[0])
            algs = os.listdir('./result/' + bench + '/' + cur_batch)
            algs = [a for a in algs if a in all_algs]
            if not os.path.exists("../PFD_results/" + bench + '-' + cur_batch):
                os.makedirs("./PFD_results/" + bench + '-' + cur_batch, exist_ok=True)
            All_outputs = {}
            for cur_alg in algs:
                path = './result/' + bench + '/' + cur_batch + '/' + cur_alg
                seed_cnt = os.listdir(path)
                seed_cnt = [s for s in seed_cnt if not s.startswith('.')]
                seed_cnt = sorted(seed_cnt, key=int)
                vals = []
                for cur_seed in seed_cnt:
                    res_path = path + '/' + cur_seed + "/EvaluatedSamples.csv"
                    t = read_csv(res_path)
                    outputs = [[float(t[p][1 + n_var + qq]) for qq in range(n_obj)] for p in range(1, len(t))]
                    vals.append(outputs)
                All_outputs[cur_alg] = vals
            all_methods = list(All_outputs.keys())
            limits = [[All_outputs[all_methods[0]][0][0][z], All_outputs[all_methods[0]][0][0][z]] for z in range(n_obj)]
            for i in all_methods:
                for j in range(len(All_outputs[i])):
                    for k in range(len(All_outputs[i][j])):
                        for l in range(len(All_outputs[i][j][k])):
                            if All_outputs[i][j][k][l] < limits[l][0]:
                                limits[l][0] = All_outputs[i][j][k][l]
                            if All_outputs[i][j][k][l] > limits[l][1]:
                                limits[l][1] = All_outputs[i][j][k][l]
            for cur_key in all_methods:
                diversities = calculate_diversity_by_seed(All_outputs[cur_key], batch_size, True, limits)
                path_mean = "./PFD_results/" + bench + '-' + cur_batch + "/" + cur_key + "-mean_PFD.txt"
                path_std = "./PFD_results/" + bench + '-' + cur_batch + "/" + cur_key + "-std_PFD.txt"
                f_mean_divs = [np.mean(sub_list) for sub_list in zip(*diversities)]
                f_std_divs = [np.std(sub_list) for sub_list in zip(*diversities)]
                write_to_file(path_mean, f_mean_divs)
                write_to_file(path_std, f_std_divs)
            if ALL_LIM == []:
                ALL_LIM = limits
            else:
                for lim in range(n_obj):
                    ALL_LIM[lim][0] = min(ALL_LIM[lim][0], limits[lim][0])
                    ALL_LIM[lim][1] = max(ALL_LIM[lim][1], limits[lim][1])


def main():
    calculate_PFD()


if __name__ == "__main__":
    main()
