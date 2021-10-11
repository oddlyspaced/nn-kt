import supervised.Adaline
import supervised.Perceptron

fun main() {
//    println("Hello World")
//
//    val input =
//        arrayOf(
//            arrayOf(1.0, 1.0),
//            arrayOf(1.0, -1.0),
//            arrayOf(-1.0, 1.0),
//            arrayOf(-1.0, -1.0),
//        )
//    val output = arrayOf(1.0, -1.0, -1.0, -1.0)
//    val weights = arrayOf(0.0, 0.0)
//    val bias = 0.0
//
//    Perceptron(
//        input,
//        output,
//        weights,
//        bias,
//        0.0,
//        1.0
//    ).calculate(2)

//    val input =
//        arrayOf(
//            arrayOf(1.0, 1.0, 1.0, 1.0),
//            arrayOf(-1.0, 1.0, -1.0, -1.0),
//            arrayOf(1.0, 1.0, 1.0, -1.0),
//            arrayOf(1.0, -1.0, -1.0, 1.0),
//        )
//    val output = arrayOf(1.0, 1.0, -1.0, -1.0)
//    val weights = arrayOf(0.0, 0.0, 0.0, 0.0)
//    val bias = 0.0
//
//    Perceptron(
//        input,
//        output,
//        weights,
//        bias,
//        0.2,
//        1.0
//    ).calculate(2)

    // =========

//    val input =
//        arrayOf(
//            arrayOf(1.0, 1.0),
//            arrayOf(1.0, -1.0),
//            arrayOf(-1.0, 1.0),
//            arrayOf(-1.0, -1.0),
//        )
//    val output = arrayOf(1.0, 1.0, 1.0, -1.0)
//    val weights = arrayOf(0.1, 0.1)
//    val bias = 0.1
//    val alpha = 0.1
//
//    Adaline(
//        input,
//        output,
//        weights,
//        bias,
//        alpha
//    ).calculate(1)

    val input =
        arrayOf(
            arrayOf(1.0, 1.0),
            arrayOf(1.0, -1.0),
            arrayOf(-1.0, 1.0),
            arrayOf(-1.0, -1.0),
        )
    val output = arrayOf(-1.0, 1.0, -1.0, -1.0)
    val weights = arrayOf(0.2, 0.2)
    val bias = 0.2
    val alpha = 0.2

    Adaline(
        input,
        output,
        weights,
        bias,
        alpha
    ).calculate(2)

}