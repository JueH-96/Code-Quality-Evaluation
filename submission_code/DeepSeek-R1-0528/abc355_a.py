def main():
    A, B = map(int, input().split())
    suspects = {1, 2, 3}
    suspects -= {A, B}
    print(suspects.pop() if len(suspects) == 1 else -1)

if __name__ == "__main__":
    main()