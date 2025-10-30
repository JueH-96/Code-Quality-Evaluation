def main():
    import sys
    data = sys.stdin.read().strip().split()
    if len(data) != 81:
        print("No")
        return
    nums = list(map(int, data))
    # build 9x9 grid
    A = [nums[i*9:(i+1)*9] for i in range(9)]
    # helper to check if a list of 9 ints is a permutation of 1..9
    def is_valid_group(group):
        # Using a bitmask or set
        s = set(group)
        return len(group) == 9 and s == set(range(1,10))
    # check rows
    for row in A:
        if not is_valid_group(row):
            print("No")
            return
    # check columns
    for j in range(9):
        col = [A[i][j] for i in range(9)]
        if not is_valid_group(col):
            print("No")
            return
    # check 3x3 blocks
    for bi in range(0, 9, 3):
        for bj in range(0, 9, 3):
            block = []
            for di in range(3):
                for dj in range(3):
                    block.append(A[bi+di][bj+dj])
            if not is_valid_group(block):
                print("No")
                return
    # all checks passed
    print("Yes")

if __name__ == "__main__":
    main()