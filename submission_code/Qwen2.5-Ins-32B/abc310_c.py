import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    sticks = data[1:]
    
    unique_sticks = set()
    
    for stick in sticks:
        # Add the stick or its reverse to the set
        unique_sticks.add(min(stick, stick[::-1]))
    
    print(len(unique_sticks))

if __name__ == "__main__":
    main()