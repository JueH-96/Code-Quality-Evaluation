from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    
    pos_dict = defaultdict(list)
    for idx, val in enumerate(A, start=1):
        pos_dict[val].append(idx)
    
    answer = 0
    
    for val in pos_dict:
        pos = pos_dict[val]
        m = len(pos)
        if m < 2:
            continue
        sum1 = 0
        sum2 = 0
        for i in range(m):
            term_val = pos[i] - (i + 1)
            sum1 += term_val * i
            sum2 += term_val * (m - i - 1)
        answer += (sum1 - sum2)
    
    print(answer)

if __name__ == "__main__":
    main()