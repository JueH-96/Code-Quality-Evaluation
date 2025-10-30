def main():
    import sys
    data = sys.stdin.read().strip().split()
    B = int(data[0])
    G = int(data[1])
    
    if B > G:
        print("Bat")
    else:
        print("Glove")

if __name__ == "__main__":
    main()