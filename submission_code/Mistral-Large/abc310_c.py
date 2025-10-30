import sys

def count_unique_sticks(N, sticks):
    unique_sticks = set()

    for stick in sticks:
        # Add both the original and the reversed string to the set
        unique_sticks.add(min(stick, stick[::-1]))

    return len(unique_sticks)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    sticks = data[1:]

    result = count_unique_sticks(N, sticks)
    print(result)

if __name__ == "__main__":
    main()