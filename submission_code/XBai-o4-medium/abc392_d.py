import sys
from collections import Counter

def main():
    n = int(sys.stdin.readline())
    dice = []
    for _ in range(n):
        parts = list(map(int, sys.stdin.readline().split()))
        k_i = parts[0]
        a_list = parts[1:]
        cnt = Counter(a_list)
        dice.append((k_i, cnt))
    
    max_prob = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            ki, cnti = dice[i]
            kj, cntj = dice[j]
            current_sum = 0.0
            if len(cnti) <= len(cntj):
                for x in cnti:
                    if x in cntj:
                        current_sum += cnti[x] * cntj[x]
            else:
                for x in cntj:
                    if x in cnti:
                        current_sum += cntj[x] * cnti[x]
            prob = current_sum / (ki * kj)
            if prob > max_prob:
                max_prob = prob
    
    print("{0:.15f}".format(max_prob))

if __name__ == "__main__":
    main()