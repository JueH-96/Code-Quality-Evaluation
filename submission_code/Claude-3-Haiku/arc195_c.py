from typing import List, Tuple

def solve_case(R: int, B: int) -> Tuple[str, List[Tuple[str, int, int]]]:
    if R + B < 2:
        return "No", []

    pieces = []
    for i in range(1, R + B + 1):
        if i <= R:
            pieces.append(("R", i, i))
        else:
            pieces.append(("B", i - R, i - R))

    for i in range(1, R + B):
        if pieces[i - 1][0] == "R":
            if pieces[i][0] == "R":
                pieces[i] = ("R", pieces[i][1] + 1, pieces[i][2])
            else:
                pieces[i] = ("B", pieces[i][1] + 1, pieces[i][2] - 1)
        else:
            if pieces[i][0] == "B":
                pieces[i] = ("B", pieces[i][1] + 1, pieces[i][2] + 1)
            else:
                pieces[i] = ("R", pieces[i][1] - 1, pieces[i][2] + 1)

    if pieces[0][1] == pieces[-1][1] and pieces[0][2] == pieces[-1][2]:
        return "Yes", pieces
    else:
        return "No", []

def main():
    T = int(input())
    for _ in range(T):
        R, B = map(int, input().split())
        result, placement = solve_case(R, B)
        print(result)
        if result == "Yes":
            for p, r, c in placement:
                print(f"{p} {r} {c}")

if __name__ == "__main__":
    main()