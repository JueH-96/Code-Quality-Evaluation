import math
import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    m = N * K
    log_fact = [0.0] * (m + 1)
    for i in range(2, m + 1):
        log_fact[i] = log_fact[i-1] + math.log(i)
    
    counts = {i: K for i in range(1, N+1)}
    sequence = []
    current_k = (0 + 1) // 2  # Placeholder, needs correct calculation
    
    for _ in range(m):
        sorted_numbers = sorted([num for num, cnt in counts.items() if cnt > 0])
        for i, num in enumerate(sorted_numbers):
            remaining = counts.copy()
            remaining[num] -= 1
            if remaining[num] == 0:
                del remaining[num]
            
            total_remaining = m - len(sequence) - 1
            sum_log = 0.0
            for cnt in remaining.values():
                if cnt > 0:
                    sum_log += log_fact[cnt]
            log_count = log_fact[total_remaining] - sum_log
            
            log_k = math.log(current_k) if current_k > 0 else -math.inf
            if log_count >= log_k:
                sequence.append(num)
                counts[num] -= 1
                if counts[num] == 0:
                    del counts[num]
                break
            else:
                current_k -= int(math.exp(log_count))
        else:
            pass
    
    print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()