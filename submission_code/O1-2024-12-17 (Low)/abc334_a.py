def main():
    import sys
    line = sys.stdin.readline().strip()
    B, G = map(int, line.split())
    if B > G:
        print("Bat")
    else:
        print("Glove")

main()