import sys
sys.setrecursionlimit(10000)

def main() -> None:
    # read input
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    res = set()                    # all XOR values obtained

    # depth-first enumeration of all partitions
    def dfs(idx: int, groups: list, cur_xor: int) -> None:
        """
        idx       : index of the next bag to place (0-based)
        groups    : current list with the sum in each non-empty group formed so far
        cur_xor   : XOR of all values currently in 'groups'
        """
        if idx == N:               # all bags processed – store the XOR
            res.add(cur_xor)
            return

        x = A[idx]

        # put the bag into every already existing group
        for i in range(len(groups)):
            old_sum = groups[i]
            new_sum = old_sum + x

            groups[i] = new_sum                 # update group sum
            dfs(idx + 1, groups, cur_xor ^ old_sum ^ new_sum)
            groups[i] = old_sum                 # restore for next branch

        # start a brand-new group with this bag alone
        groups.append(x)
        dfs(idx + 1, groups, cur_xor ^ x)
        groups.pop()                            # back-track

    dfs(0, [], 0)
    print(len(res))


if __name__ == "__main__":
    main()