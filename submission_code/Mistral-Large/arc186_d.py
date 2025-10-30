import sys

MOD = 998244353

def is_polish(seq, memo):
    if seq in memo:
        return memo[seq]

    n = len(seq)
    if n == 1:
        return seq[0] == 0

    v1 = seq[0]
    if v1 > n - 1:
        memo[seq] = False
        return False

    subseq_len = (n - 1) // v1
    for i in range(v1):
        start = i * subseq_len + 1
        end = start + subseq_len
        if not is_polish(tuple(seq[start:end]), memo):
            memo[seq] = False
            return False

    memo[seq] = True
    return True

def count_polish_sequences(n, a):
    memo = {}
    count = 0

    def generate_sequences(current, length):
        nonlocal count
        if length == n:
            if current <= a and is_polish(current, memo):
                count = (count + 1) % MOD
            return
        for i in range(n):
            generate_sequences(current + (i,), length + 1)

    generate_sequences((), 0)
    return count

def main():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = tuple(map(int, data[1:]))
    print(count_polish_sequences(n, a))

if __name__ == "__main__":
    main()