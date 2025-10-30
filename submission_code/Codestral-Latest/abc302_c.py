from itertools import permutations

def can_rearrange(strings):
    n = len(strings)
    m = len(strings[0])

    for perm in permutations(strings):
        valid = True
        for i in range(n - 1):
            diff_count = 0
            for j in range(m):
                if perm[i][j] != perm[i + 1][j]:
                    diff_count += 1
            if diff_count != 1:
                valid = False
                break
        if valid:
            return True
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    strings = data[2:]

    if can_rearrange(strings):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()