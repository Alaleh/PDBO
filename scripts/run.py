import os, signal
from time import time
from argparse import ArgumentParser
from get_ref_point import get_ref_point
from multiprocessing import Process, Queue, cpu_count


def worker(cmd, problem, algo, batch, seed, queue):
    ret_code = os.system(cmd)
    queue.put([ret_code, problem, algo, batch, seed])


def main():
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, nargs='+', required=True, help='problems to test')
    parser.add_argument('--algo', type=str, nargs='+', default='PDBO', required=True, help='algorithms to test')
    parser.add_argument('--n-seed', type=int, default=25, help='number of different seeds')
    parser.add_argument('--n-process', type=int, default=cpu_count(), help='number of parallel optimization executions')
    parser.add_argument('--n-inner-process', type=int, default=1, help='number of process can be used for each optimization')
    parser.add_argument('--subfolder', type=str, default='default', help='subfolder of result')
    parser.add_argument('--exp-name', type=str, default=None, help='custom experiment name')
    parser.add_argument('--batch-size', type=str, nargs='+', required=True, help='batch sizes to test')
    parser.add_argument('--n-iter', type=int)
    parser.add_argument('--n-var', type=int)
    parser.add_argument('--n-obj', type=int)
    args = parser.parse_args()

    # get reference point for each problem first, make sure every algorithm and every seed run using the same reference point
    ref_dict = {}
    for problem in args.problem:
        ref_point = get_ref_point(problem, args.n_var, args.n_obj)
        ref_point_str = ' '.join([str(val) for val in ref_point])
        ref_dict[problem] = ref_point_str

    queue = Queue()
    n_active_process = 0
    start_time = time()
    iter_cnt = {1: 250, 2: 125, 4: 64, 8: 32, 16: 16, 32: 8}

    for batch_s in args.batch_size:
        for seed in range(args.n_seed):
            for problem in args.problem:
                for algo in args.algo:

                    args.subfolder = batch_s+"-batch"

                    command = f'python -W ignore main.py \
                        --problem {problem} --algo {algo} --seed {seed} \
                        --batch-size {int(batch_s)} --n-iter {iter_cnt[int(batch_s)]} \
                        --ref-point {ref_dict[problem]} \
                        --n-process {args.n_inner_process} \
                        --subfolder {batch_s+"-batch"} --log-to-file'

                    command += f' --n-var {args.n_var} --n-obj {args.n_obj}'

                    if args.exp_name is not None:
                        command += f' --exp-name {args.exp_name}'

                    Process(target=worker, args=(command, problem, algo, batch_s, seed, queue)).start()
                    print(f'problem {problem} algo {algo} batch size {batch_s} seed {seed} started')
                    n_active_process += 1

                    if n_active_process >= args.n_process:
                        ret_code, ret_problem, ret_algo, ret_batch_s, ret_seed = queue.get()
                        if ret_code == signal.SIGINT:
                            exit()
                        print(f'problem {ret_problem} algo {ret_algo} batch size {ret_batch_s} seed {ret_seed} done, time: ' + '%.2fs' % (time() - start_time))
                        n_active_process -= 1
    
    for _ in range(n_active_process):
        ret_code, ret_problem, ret_algo, ret_batch_s, ret_seed = queue.get()
        if ret_code == signal.SIGINT:
            exit()
        print(f'problem {ret_problem} algo {ret_algo} batch size {ret_batch_s} seed {ret_seed} done, time: ' + '%.2fs' % (time() - start_time))

    print('all experiments done, time: %.2fs' % (time() - start_time))
    

if __name__ == "__main__":
    main()
