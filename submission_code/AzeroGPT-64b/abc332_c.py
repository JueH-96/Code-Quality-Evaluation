from typing import List

def tshirt_planner(N: int, M: int, S: str) -> int:
    """
    Determine the minimum number of AtCoder logo T-shirts Takahashi needs to buy to be able to wear
    appropriate T-shirts on all scheduled days during the N days, given the conditions.
    
    Parameters:
    N (int): The number of days in Takahashi's schedule.
    M (int): The number of plain T-shirts Takahashi has.
    S (str): A string representing Takahashi's schedule.
    
    Returns:
    int: The minimum number of T-shirts Takahashi needs to buy.
    """
    # Count the number of days Takahashi plans to attend a competitive programming event.
    events = S.count('2')
    
    # If Takahashi has enough plain T-shirts to cover all non-event days, calculate the extra T-shirts needed.
    if M >= N - events:
        return max(0, events - M)
    
    # If not, calculate the extra T-shirts needed by considering both event and non-event days.
    T = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        if S[i] in '12':
            T[i] = 1 + (T[i + 1] if S[i] == '1' else T[i + M])
    
    return T[0] - (M < events)

# Reading inputs
if __name__ == "__main__":
    N, M = map(int, input().split())
    S = input()
    print(tshirt_planner(N, M, S))