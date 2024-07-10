alert("This Clone of home page of RIT made only for learning purpose !")
class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, chatBox, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))

        sendButton.addEventListener('click', () => this.onSendButton(chatBox))

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "Sam", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox)
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox)
            textField.value = ''
          });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam")
            {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}


const chatbox = new Chatbox();
chatbox.display();

















let slideIndex = 0;
showSlides();

// Next/previous controls
function plusSlides(n) {
  showSlides(slideIndex += n);
}

// Thumbnail image controls
function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides() {
  let i;
  let slides = document.getElementsByClassName("mySlide");
  let dots = document.getElementsByClassName("dot");
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  slideIndex++;
  if (slideIndex > slides.length) {
    slideIndex = 1;
  }
  slides[slideIndex - 1].style.display = "block";
  if (dots.length > 0) {
    for (i = 0; i < dots.length; i++) {
      dots[i].className = dots[i].className.replace(" active", "");
    }
    dots[slideIndex - 1].className += " active";
  }
  setTimeout(showSlides, 3000); // Change image every 3 seconds
}




window.addEventListener('scroll', function() {
    var header = document.querySelector('.head');
    if (window.scrollY > 0) {
        header.classList.add('sticky');
    } else {
        header.classList.remove('sticky');
    }
});





var slideIndex1 = 1;
showSlides1(slideIndex1);

function plusSlides1(n) {
    showSlides1(slideIndex1 += n);
}

function currentSlide1(n) {
    showSlides1(slideIndex1 = n);
}

function showSlides1(n) {
    var i;
    var slides1 = document.getElementsByClassName("mySlides-1");
    var dots1 = document.getElementsByClassName("dot-1");
    if (n > slides1.length) {slideIndex1 = 1}
    if (n < 1) {slideIndex1 = slides1.length}
    for (i = 0; i < slides1.length; i++) {
        slides1[i].style.display = "none";
    }
    for (i = 0; i < dots1.length; i++) {
        dots1[i].className = dots1[i].className.replace(" active-1", "");
    }
    slides1[slideIndex1-1].style.display = "block";
    dots1[slideIndex1-1].className += " active-1";
}



var swiper = new Swiper(".mySwiper", {
  slidesPerView: 1,
  grabCursor: true,
  loop: true,
  pagination: {
      el: ".swiper-pagination",
      clickable: true,
  },
  navigation: {
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev",
  },
});