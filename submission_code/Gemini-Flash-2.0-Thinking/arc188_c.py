def solve():
    n, m = map(int, input().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, input().split())))

    for i in range(2**n):
        confused_mask = bin(i)[2:].zfill(n)
        confused = [int(c) for c in confused_mask]

        def is_consistent_honesty(honest):
            for a, b, c_val in testimonies:
                a -= 1
                b -= 1

                a_honest = honest[a]
                b_honest = honest[b]
                a_confused = confused[a]

                teller_truthful = (a_honest == 1 and a_confused == 0) or (a_honest == 0 and a_confused == 1)

                testimony_matches = False
                if c_val == 0:  # A says B is honest
                    testimony_matches = (b_honest == 1)
                else:  # A says B is a liar
                    testimony_matches = (b_honest == 0)

                if teller_truthful != testimony_matches:
                    return False
            return True

        found_consistent = False
        for j in range(2**n):
            honest_mask = bin(j)[2:].zfill(n)
            honest = [int(h) for h in honest_mask]
            if is_consistent_honesty(honest):
                found_consistent = True
                break

        if found_consistent:
            print(confused_mask)
            return

    print("-1")

if __name__ == "__main__":
    solve()