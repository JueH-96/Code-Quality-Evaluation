MOD = 998244353

def solve():
    n, m, k = map(int, input().split())
    x = list(map(int, input().split()))

    count = 0
    
    def is_subsequence(a, b):
        i = 0
        j = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
            j += 1
        return i == len(a)

    def generate_sequences(length, current_sequence):
        nonlocal count
        if length == n:
            valid = True
            for i in range(k**m):
                seq = []
                temp = i
                for _ in range(m):
                    seq.append(temp % k + 1)
                    temp //= k
                seq.reverse()
                if seq == x:
                    continue
                if is_subsequence(seq, current_sequence):
                    continue
                else:
                    valid = False
                    break
            if valid:
                count = (count + 1) % MOD
            return

        for i in range(1, k + 1):
            generate_sequences(length + 1, current_sequence + [i])

    generate_sequences(0, [])
    print(count)

solve()