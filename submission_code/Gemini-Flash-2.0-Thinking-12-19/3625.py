class Solution:
    def canAliceWin(self, n: int) -> bool:
        if n < 10:
            return False
        n -= 10
        remove_bob = 9
        remove_alice = 8
        turn = "Bob"
        while True:
            if turn == "Bob":
                if n < remove_bob:
                    return True
                n -= remove_bob
                turn = "Alice"
                remove_alice = remove_bob - 1
            else:
                if n < remove_alice:
                    return False
                n -= remove_alice
                turn = "Bob"
                remove_bob = remove_alice - 1
            if remove_bob < 1 or remove_alice < 1:
                if turn == "Bob":
                    return True
                else:
                    return False