## Что это такое
Развивающийся язык программирования ShittyAScript  
### Почему
Потому что

### Запуск
Создание файла с расширением .shittyass  
Запуск ```myshit.py <Имя файла>``` 

### Синтаксис
Язык начинается с объявления переменных в глобальном скоупе  
Далее следуют выражения  
Выражения могут быть либо строкой вида `rhs = lhs`, либо созданием нового скоупа,  
**в таком случае объявление начинается сначала**  
Выражения так же поддерживают (пока один) кейворды:  
print  
Например,  
`print = '123'`
>123

Выражения поддерживают базовую арифметику: сложение, вычитание, умножение и деление (только целочисленное)  
Для использования применяется специальный оператор `@=`, например,  
`print @= 123 + 4 / 2`
> 125  

Выражения поддерживают return (возврат значения) из вложенных скоупов,  
и разыменовывание результата через специальную переменную `return`, например,  
``` 
ShitScript = 11
OhShit
{
    OhShit
    return @= 25 * 10 + 50
}
ShitScript = return
print = ShitScript
EndShit
```
> 300  

Можно сделать смешно
```
OhShit
print = 1
print = print
EndShit
```
>1  
>1

Можно делать функции, функции объявляются ключевым словом SHIT (case-sensitive),  
вызов функции производится при помощи символа %, например,  

```
func SHIT (n,balls) {
    print = n
    print = balls
    return @= n + balls
    EndShit
}

%func(2,4)
print = return
%func(1,10)
print = return
EndShit
```
>2  
>4  
>6  
>1  
>10  
>11  



