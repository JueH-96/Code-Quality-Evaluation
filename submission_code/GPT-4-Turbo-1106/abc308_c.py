def main():
    N = int(input())
    people = []
    for i in range(N):
        A, B = map(int, input().split())
        success_rate = A / (A + B)
        people.append((success_rate, i + 1))
    people.sort(key=lambda x: (-x[0], x[1]))
    for _, person_number in people:
        print(person_number, end=' ')

if __name__ == "__main__":
    main()