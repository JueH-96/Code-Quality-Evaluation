def count_arithmetic_subarrays(N, A):
    if N == 1:
        return 1
    D = []
    for i in range(1, N):
        D.append(A[i] - A[i-1])
    count = 0
    current_group_length = 1
    for i in range(1, len(D)):
        if D[i] == D[i-1]:
            current_group_length += 1
        else:
            count += current_group_length * (current_group_length + 1) // 2
            current_group_length = 1
    count += current_group_length * (current_group_length + 1) // 2  # for the last group
    total_count = count + N
    return total_count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    print(count_arithmetic_subarrays(N, A))

if __name__ == "__main__":
    main()