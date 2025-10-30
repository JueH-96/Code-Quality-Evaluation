def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    def get_pyramid_sequence(k):
        if k <= 0:
            return []
        sequence = []
        for i in range(1, k):
            sequence.append(i)
        sequence.append(k)
        for i in range(k - 1, 0, -1):
            sequence.append(i)
        return sequence

    max_k_possible = 0
    max_possible_size = (n + 1) // 2
    
    for k in range(max_possible_size, 0, -1):
        pyramid = get_pyramid_sequence(k)
        pyramid_len = len(pyramid)
        if pyramid_len > n:
            continue
        found_subsequence = False
        for l in range(n - pyramid_len + 1):
            subsequence = a[l:l+pyramid_len]
            possible_pyramid = True
            for i in range(pyramid_len):
                if subsequence[i] < pyramid[i]:
                    possible_pyramid = False
                    break
            if possible_pyramid:
                found_subsequence = True
                break
        if found_subsequence:
            max_k_possible = k
            break
            
    if max_k_possible > 0:
        print(max_k_possible)
    else:
        print(1)

if __name__ == '__main__':
    solve()