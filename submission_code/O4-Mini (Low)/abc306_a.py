def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    s = data[1]
    result = []
    for ch in s:
        result.append(ch * 2)
    sys.stdout.write(''.join(result))

if __name__ == "__main__":
    main()