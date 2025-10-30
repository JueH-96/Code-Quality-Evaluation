import sys

def main():
    # Read N and D from the first line of input.
    # N is the total number of boxes.
    # D is the number of days (and thus cookies eaten).
    N, D = map(int, sys.stdin.readline().split())
    
    # Read the string S representing the state of the boxes.
    S = sys.stdin.readline().strip()
    
    # Count the number of boxes that are initially empty.
    # An empty box is represented by '.'
    initial_empty_count = S.count('.')
    
    # Over D days, D cookies are eaten. Each eaten cookie results in one
    # box (that previously had a cookie) becoming empty.
    # These D boxes are distinct from the ones that were initially empty.
    # So, the total number of empty boxes after D days is the sum of
    # initially empty boxes and the D boxes that become empty.
    num_empty_after_D_days = initial_empty_count + D
    
    # Print the final count of empty boxes.
    print(num_empty_after_D_days)

if __name__ == '__main__':
    main()