import sys

def main():
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    strings = data[1:N+1]
    count = strings.count("Takahashi")
    print(count)

if __name__ == "__main__":
    main()