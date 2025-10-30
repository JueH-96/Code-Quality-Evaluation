class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # If we have more 'AA' than 'BB', we start and end with 'AA'
        # If we have more 'BB' than 'AA', we start and end with 'BB'
        # If 'AA' and 'BB' are equal, we can start with either 'AA' or 'BB'
        # The 'AB' strings can be used to separate 'AA' and 'BB' when they are in excess
        # The maximum length is twice the sum of the minimum of 'AA' and 'BB' plus 'AB' strings
        # plus twice the difference between 'AA' and 'BB' if 'AB' strings are enough to separate them
        # plus one more 'AA' or 'BB' if there is an extra 'AB' to separate them
        
        # If we have more 'AB' than we need to separate 'AA' and 'BB', we can use the extra 'AB' to add one more 'AA' or 'BB'
        if z >= abs(x - y):
            return 2 * (x + y + min(z, x + y))
        # If we don't have enough 'AB', we can only add as many 'AA' or 'BB' as we have 'AB' to separate them
        else:
            return 2 * (min(x, y) + z) + max(x, y)