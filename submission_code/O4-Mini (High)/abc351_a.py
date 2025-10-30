def main():
    # Read runs scored by Team Takahashi in innings 1 through 9
    A = list(map(int, input().split()))
    # Read runs scored by Team Aoki in innings 1 through 8
    B = list(map(int, input().split()))
    # To win, Team Aoki needs to outscore Team Takahashi:
    # sum(B) + x > sum(A)  =>  x > sum(A) - sum(B)
    # minimal integer x = (sum(A) - sum(B)) + 1
    needed = sum(A) - sum(B) + 1
    print(needed)

main()