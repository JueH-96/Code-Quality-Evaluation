import math

def solve():
    n, k = map(int, input().split())
    nk = n * k
    target_rank = 0
    
    def get_total_good_sequences_count(current_counts):
        remaining_length = 0
        counts_to_use = []
        for i in range(n):
            count = k - current_counts[i]
            if count < 0:
                return 0
            counts_to_use.append(count)
            remaining_length += count
        if remaining_length < 0:
            return 0
        if remaining_length == 0:
            return 1
        
        numerator = math.factorial(remaining_length)
        denominator = 1
        for count in counts_to_use:
            denominator *= math.factorial(count)
        return numerator // denominator

    total_sequences = get_total_good_sequences_count([0] * n)
    target_rank = (total_sequences + 1) // 2
    
    result_sequence = []
    counts_used = [0] * n
    current_rank = target_rank
    
    for _ in range(nk):
        for number in range(1, n + 1):
            if counts_used[number-1] < k:
                counts_used[number-1] += 1
                sequences_count = get_total_good_sequences_count(counts_used)
                counts_used[number-1] -= 1
                if current_rank <= sequences_count:
                    result_sequence.append(number)
                    counts_used[number-1] += 1
                    break
                else:
                    current_rank -= sequences_count
                    
    print(*(result_sequence))

if __name__ == '__main__':
    solve()