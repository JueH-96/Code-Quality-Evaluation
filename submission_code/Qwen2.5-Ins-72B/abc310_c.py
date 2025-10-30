# YOUR CODE HERE
import sys

def count_unique_sticks(N, sticks):
    unique_sticks = set()
    for stick in sticks:
        # Add both the stick and its reverse to the set
        unique_sticks.add(min(stick, stick[::-1]))
    return len(unique_sticks)

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    sticks = data[1:]
    print(count_unique_sticks(N, sticks))