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

    def is_possible(k):
        pyramid = get_pyramid_sequence(k)
        if len(pyramid) > n:
            return False
        pyramid_len = len(pyramid)
        for start_index in range(n - pyramid_len + 1):
            possible_subsequence = True
            for i in range(pyramid_len):
                if a[start_index + i] < pyramid[i]:
                    possible_subsequence = False
                    break
            if possible_subsequence:
                return True
        return False

    max_size = 0
    for k in range(1, n + 1):
        if is_possible(k):
            max_size = max(max_size, k)
            
    if n == 1:
        if a[0] >= 1:
            print(1)
        else:
            print(0)
        return
        
    best_size = 0
    for k_test in range(1, n // 2 + 2):
        if is_possible(k_test):
            best_size = max(best_size, k_test)
            
    if best_size == 0 and n >= 1:
        if is_possible(1):
            best_size = 1

    print(best_size)

if __name__ == '__main__':
    solve()