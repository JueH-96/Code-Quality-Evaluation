import sys

def calculate_sum(N, A):
    total_sum = 0
    for i in range(N-1):
        for j in range(i+1, N):
            max_val = max(A[i], A[j])
            min_val = min(A[i], A[j])
            total_sum += max_val // min_val
    return total_sum

def main():
    input_lines = sys.stdin.readlines()
    N = int(input_lines[0].strip())
    A = list(map(int, input_lines[1].strip().split()))
    result = calculate_sum(N, A)
    print(result)

if __name__ == "__main__":
    main()