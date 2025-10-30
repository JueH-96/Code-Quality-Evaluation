import sys
from math import factorial
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    total_length = N * K
    freq = [K] * N
    total = total_length
    
    def compute_initial_S():
        num = 1
        for i in range(1, total_length + 1):
            num *= i
        denom = factorial(K) ** N
        S_val = num // denom
        return S_val

    S_state = compute_initial_S()
    M = (S_state + 1) // 2
    current_count = 0
    result = []
    
    for _ in range(total_length):
        found = False
        for d in range(1, N + 1):
            idx = d - 1
            if freq[idx] == 0:
                continue
            count_val = (S_state * freq[idx]) // total
            if current_count + count_val >= M:
                result.append(str(d))
                freq[idx] -= 1
                total -= 1
                S_state = count_val
                if total > 0:
                    M -= current_count
                found = True
                break
            else:
                current_count += count_val
        if not found:
            for d in range(1, N + 1):
                if freq[d - 1] > 0:
                    result.append(str(d))
                    freq[d - 1] -= 1
                    total -= 1
                    break
    print(" ".join(result))

if __name__ == "__main__":
    main()