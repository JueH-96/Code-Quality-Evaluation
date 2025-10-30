import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    count_list = []
    k_list = []
    for _ in range(n):
        tokens = list(map(int, sys.stdin.readline().split()))
        k = tokens[0]
        nums = tokens[1:]
        count = Counter(nums)
        count_list.append(count)
        k_list.append(k)
    
    max_prob = 0.0
    for i in range(n):
        count_i = count_list[i]
        ki = k_list[i]
        for j in range(i + 1, n):
            count_j = count_list[j]
            kj = k_list[j]
            sum_common = 0.0
            # Choose the smaller counter to iterate
            if len(count_i) <= len(count_j):
                for x in count_i:
                    if x in count_j:
                        sum_common += count_i[x] * count_j[x]
            else:
                for x in count_j:
                    if x in count_i:
                        sum_common += count_i[x] * count_j[x]
            current_prob = sum_common / (ki * kj)
            if current_prob > max_prob:
                max_prob = current_prob
    print("{0:.15f}".format(max_prob))

if __name__ == "__main__":
    main()