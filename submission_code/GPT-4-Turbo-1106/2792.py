class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        # The XOR of all elements in derived must be 0 for a valid original to exist.
        # This is because in a circular XOR chain, all 1's would cancel out.
        return all(x == 0 for x in derived) or derived[0] == derived[-1]