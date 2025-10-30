def main():
    n = int(input().strip())
    s = input().strip()
    max_run = {}
    i = 0
    n_len = len(s)
    while i < n_len:
        j = i
        while j < n_len and s[j] == s[i]:
            j += 1
        length = j - i
        c = s[i]
        if c in max_run:
            if length > max_run[c]:
                max_run[c] = length
        else:
            max_run[c] = length
        i = j

    total = sum(max_run.values())
    print(total)

if __name__ == "__main__":
    main()