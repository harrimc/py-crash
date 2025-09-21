def get_water_bill(gallons) :
    if gallons <= 8000 :
        price = ((gallons/1000)*5)
    elif gallons <= 22000 :
        price = ((gallons/1000)*6)
    elif gallons <= 30000 :
        price = ((gallons/1000)*7)
    else :
        price = ((gallons/1000)*10)
    return price

print('How much water did you use?') 
gall = float(input('> '))

print('your water bill is', get_water_bill(gall))