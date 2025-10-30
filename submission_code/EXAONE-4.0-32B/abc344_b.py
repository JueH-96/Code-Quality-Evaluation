import sys

def main():
    data = []
    for line in sys.stdin:
        stripped = line.strip()
        if stripped:
            data.append(int(stripped))
    
    for num in reversed(data):
        print(num)

if __name__ == "__main__":
    main()