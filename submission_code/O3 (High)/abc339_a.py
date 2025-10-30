import sys

def main():
    # Read the single input line and strip the trailing newline
    s = sys.stdin.readline().rstrip('
')
    
    # Split by '.' and output the last part
    print(s.split('.')[-1])

if __name__ == "__main__":
    main()