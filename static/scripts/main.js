function HighlightNavbar() {
  const path = window.location.href.split('/')
  let found = false
  let i
  for (i = 0; i < path.length; i++) {
    switch (path[i].split('_')[0]) {
      case 'home':
        document.getElementById('nav-home').classList.add('active')
        found = true
        break
      case 'predict':
        document.getElementById('nav-predict').classList.add('active')
        found = true
        break
      case 'examples':
        document.getElementById('nav-examples').classList.add('active')
        found = true
        break
      case 'about':
        document.getElementById('nav-about').classList.add('active')
        found = true
        break
      case 'contact':
        document.getElementById('nav-contact').classList.add('active')
        found = true
        break
    }
  }

  if (!found) {
    document.getElementById('nav-home').classList.add('active')
  }
}

function GetFilename() {
  const x = document.getElementById('entry_value')
  console.log(x.value)
  document.getElementById('fileName').innerHTML = x.value.split('\\').pop()
}
