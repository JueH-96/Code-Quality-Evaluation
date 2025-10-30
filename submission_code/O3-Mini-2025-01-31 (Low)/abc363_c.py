def main():
    import sys
    from collections import Counter
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()

    freq = Counter(S)
    count = 0

    # Backtracking function that builds distinct permutations.
    # We also do an on-the-fly check to prune any branch that creates
    # a substring of length K that is a palindrome.
    def backtrack(current, freqs, length):
        nonlocal count
        if length == N:
            count += 1
            return
        for ch in list(freqs.keys()):
            if freqs[ch] > 0:
                new_string = current + ch
                if len(new_string) >= K:
                    substring = new_string[-K:]
                    if substring == substring[::-1]:
                        continue
                freqs[ch] -= 1
                backtrack(new_string, freqs, length + 1)
                freqs[ch] += 1

    backtrack("", freq, 0)
    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()