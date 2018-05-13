document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector("#check-quiz")
  console.log(button)
  button.addEventListener('click', () => {
    let i = 0
    let n_correct = 0
    while(true) {
      const question = document.querySelector("#question" + i)
      if (!question) {
        break
      }

      const answer = question.querySelector("span")
      const select = question.querySelector("select")
      answer.style.visibility = "visible"

      if (answer.textContent === select.value) {
        n_correct++
        answer.style.color = "Green"
      } else {
        answer.style.color = "Red"
      }

      i++
    }
    console.log(n_correct)
    const result = document.querySelector("#result")
    result.textContent = "You got " + n_correct + "/" + (i) + " correct!"
  })
})
