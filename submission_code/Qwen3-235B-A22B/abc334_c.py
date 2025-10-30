import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+K]))
    
    if K == 0:
        print(0)
        return
    
    prefix = [0] * (K + 1)
    for i in range(2, K + 1):
        if i % 2 == 0:
            prefix[i] = prefix[i-2] + (A[i-1] - A[i-2])
        else:
            prefix[i] = prefix[i-1]
    
    suffix = [0] * (K + 2)
    for i in range(K-1, -1, -1):
        if i + 1 < K:
            suffix[i] = (A[i+1] - A[i]) + suffix[i+2]
        else:
            suffix[i] = 0
    
    if K % 2 == 0:
        print(prefix[K])
    else:
        min_sum = float('inf')
        for i in range(K):
            left_len = i
            if left_len % 2 == 0:
                left_sum = prefix[left_len]
                left_exists = False
            else:
                left_sum = prefix[left_len - 1]
                left_exists = True
            
            right_len = K - i - 1
            if right_len % 2 == 0:
                right_sum = suffix[i+1]
                right_exists = False
            else:
                right_sum = suffix[i+2]
                right_exists = True
            
            current_sum = left_sum + right_sum
            if left_exists and right_exists:
                current_sum += A[i+1] - A[i-1]
            
            if current_sum < min_sum:
                min_sum = current_sum
        print(min_sum)

if __name__ == '__main__':
    main()