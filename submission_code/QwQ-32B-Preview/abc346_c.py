import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    A = []
    for _ in range(N):
        A.append(int(data[idx]))
        idx += 1
    # Calculate sum from 1 to K
    sum_total = K * (K + 1) // 2
    # Use a set to store unique A_i within 1 to K
    unique_A = set()
    for num in A:
        if 1 <= num <= K:
            unique_A.add(num)
    # Calculate sum of unique A_i within 1 to K
    sum_A = sum(unique_A)
    # Calculate the answer
    answer = sum_total - sum_A
    print(answer)

if __name__ == '__main__':
    main()