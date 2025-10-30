class Solution:
	def convertDateToBinary(self, date: str) -> str:
		year_str, month_str, day_str = date.split('-')
		year_bin = bin(int(year_str))[2:]
		month_bin = bin(int(month_str))[2:]
		day_bin = bin(int(day_str))[2:]
		return f"{year_bin}-{month_bin}-{day_bin}"