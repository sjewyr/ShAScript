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

Выражения поддерживают базовую арифметику: сложение(+конкатенация), вычитание, умножение и деление (только целочисленное)  
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

**Переназначать аргументы внутри функции (shadowing) и использовать пробелы в скобках (как при вызове так и при объявлении) нельзя, к примеру**  
```
...
func SHIT (abc) {
    abc = 123
    return = abc
}
...
```
**Код в примере будет возвращать значение аргумента abc и проигнорирует ваше переназначение!**

Можно делать ветвления, синтаксис очень прост,  
```
if val susin expr { 
    ...
} else {
    ...
},
где val - переменная, число, или строка, expr - любое валидное выражение
```
Блок else может быть опущен, например,

```
balls = '11'

if balls susin '1' + '1' {
    print = 1
}
else {
    print = 2 
}

EndShit
```
>1  

**ВНИМАНИЕ! Строки и числа (например '1' и 1) не считаются равными при сравнении susin**

