import sys

def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    S = input_data[1]

    t_wins = S.count('T')
    a_wins = N - t_wins   # since every game is won by either T or A

    # Rule 1: the one with more wins is the overall winner
    if t_wins > a_wins:
        print('T')
        return
    if a_wins > t_wins:
        print('A')
        return

    # Rule 2: same number of wins – who reached that number first?
    target = t_wins  # final wins for both (they are equal here)
    t_cnt = a_cnt = 0
    for c in S:
        if c == 'T':
            t_cnt += 1
            if t_cnt == target:
                print('T')
                return
        else:  # c == 'A'
            a_cnt += 1
            if a_cnt == target:
                print('A')
                return

if __name__ == "__main__":
    main()