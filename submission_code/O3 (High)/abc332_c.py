import sys

def main() -> None:
    # Read input
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    max_logo_needed = 0          # answer to output
    ones = twos = 0              # counters inside the current block (no washing day)

    # iterate through the string and add a sentinel '0' to flush the last block
    for c in S + '0':
        if c == '0':
            # end of the current block – compute how many logo shirts are needed here
            # we can use at most M plain shirts for the '1'-days
            plain_used = min(ones, M)
            logo_needed = twos + (ones - plain_used)   # remaining '1' plus every '2'
            max_logo_needed = max(max_logo_needed, logo_needed)

            # start a new block
            ones = twos = 0
        elif c == '1':
            ones += 1
        else:             # c == '2'
            twos += 1

    print(max_logo_needed)

if __name__ == "__main__":
    main()