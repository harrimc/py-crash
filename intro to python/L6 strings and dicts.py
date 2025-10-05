def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    if len(str(zip_code)) == 5 and str(zip_code).isdigit() :
        return True 
    return False 

print(is_valid_zip('12345'))

