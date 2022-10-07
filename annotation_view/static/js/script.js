indexes = { "lymph": [], "other": [], "idk": [] }

function create_image(img64) {
    const domImg = document.createElement("img")
    domImg.src = "data:image/png;base64," + img64

    return domImg
}

function downloadObjectAsJson() {
    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(indexes));
    var downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "result" + ".json");
    document.body.appendChild(downloadAnchorNode); // required for firefox
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}


function lymph(data, indexes, count) {
    indexes.lymph.push(data.index[count])
    var parent = document.getElementById("galery_figure")
    parent.innerHTML = ""
    count++

    let image = create_image(data.images[count])
    //let image_nuc = create_image(data.images_nuc[count])
    parent.appendChild(image)
    //parent.appendChild(image_nuc)

    const caption = document.createElement("figcaption")
    caption.textContent = "Area: " + data.area[count] + " Hovernet Pred: " + data.hovernet_pred[count]

    parent.appendChild(caption)
    console.log(data.index[count], data.area[count])


    const btn_yes = document.createElement("a")
    btn_yes.textContent = "YES"
    btn_yes.style = "margin-left: -3px"
    btn_yes.classList.add("btn")
    parent.appendChild(btn_yes)

    const btn_idk = document.createElement("a")
    btn_idk.textContent = "???"
    btn_idk.classList.add("btn")
    btn_idk.style = "margin-left: 90px"

    parent.appendChild(btn_idk)
    btn_idk.addEventListener("click", (evt) => idk(data, indexes, count))


    const btn_no = document.createElement("a")
    btn_no.textContent = "NO "
    btn_no.classList.add("btn")
    btn_no.style = "margin-left: 90px"
    parent.appendChild(btn_no)
    console.log(indexes)

    btn_yes.addEventListener("click", (evt) => lymph(data, indexes, count))
    btn_no.addEventListener("click", (evt) => other(data, indexes, count))


}
function other(data, indexes, count) {
    indexes.other.push(data.index[count])
    var parent = document.getElementById("galery_figure")
    parent.innerHTML = ""
    count++

    let image = create_image(data.images[count])
    // let image_nuc = create_image(data.images_nuc[count])
    parent.appendChild(image)
    //parent.appendChild(image_nuc)

    const caption = document.createElement("figcaption")
    caption.textContent = "Area: " + data.area[count] + " Hovernet Pred: " + data.hovernet_pred[count]

    parent.appendChild(caption)
    console.log(data.index[count], data.area[count])


    const btn_yes = document.createElement("a")
    btn_yes.textContent = "YES"
    btn_yes.style = "margin-left: -3px"
    btn_yes.classList.add("btn")
    parent.appendChild(btn_yes)

    const btn_idk = document.createElement("a")
    btn_idk.textContent = "???"
    btn_idk.classList.add("btn")
    btn_idk.style = "margin-left: 90px"

    parent.appendChild(btn_idk)
    btn_idk.addEventListener("click", (evt) => idk(data, indexes, count))


    const btn_no = document.createElement("a")
    btn_no.textContent = "NO "
    btn_no.classList.add("btn")
    btn_no.style = "margin-left: 90px"
    parent.appendChild(btn_no)
    console.log(indexes)

    btn_yes.addEventListener("click", (evt) => lymph(data, indexes, count))
    btn_no.addEventListener("click", (evt) => other(data, indexes, count))

}
function idk(data, indexes, count) {
    indexes.idk.push(data.index[count])
    var parent = document.getElementById("galery_figure")
    parent.innerHTML = ""
    count++

    let image = create_image(data.images[count])
    //let image_nuc = create_image(data.images_nuc[count])
    parent.appendChild(image)
    //parent.appendChild(image_nuc)

    const caption = document.createElement("figcaption")
    caption.textContent = "Area: " + data.area[count] + " Hovernet Pred: " + data.hovernet_pred[count]

    parent.appendChild(caption)
    console.log(data.index[count], data.area[count])


    const btn_yes = document.createElement("a")
    btn_yes.textContent = "YES"
    btn_yes.style = "margin-left: -3px"
    btn_yes.classList.add("btn")
    parent.appendChild(btn_yes)

    const btn_idk = document.createElement("a")
    btn_idk.textContent = "???"
    btn_idk.classList.add("btn")
    btn_idk.style = "margin-left: 90px"

    parent.appendChild(btn_idk)
    btn_idk.addEventListener("click", (evt) => idk(data, indexes, count))


    const btn_no = document.createElement("a")
    btn_no.textContent = "NO "
    btn_no.classList.add("btn")
    btn_no.style = "margin-left: 90px"
    parent.appendChild(btn_no)
    console.log(indexes)

    btn_yes.addEventListener("click", (evt) => lymph(data, indexes, count))
    btn_no.addEventListener("click", (evt) => other(data, indexes, count))

}
function run() {
    //document.getElementById("main").style.display = "none"
    const value = { "file_name": document.getElementById("filename").value, }
    const param = new URLSearchParams(value)
    fetch("/ft?" + param).then((resp) => resp.json()).then(function (data) {
        var parent = document.getElementById("galery_figure")
        parent.innerHTML = ""
        var count = 0


        let image = create_image(data.images[count])
        //let image_nuc = create_image(data.images_nuc[count])
        parent.appendChild(image)
        //parent.appendChild(image_nuc)

        const caption = document.createElement("figcaption")
        caption.textContent = "Area: " + data.area[count] + " Hovernet Pred: " + data.hovernet_pred[count]

        parent.appendChild(caption)
        console.log(data.index[count], data.area[count])


        const btn_yes = document.createElement("a")
        btn_yes.textContent = "YES"
        btn_yes.classList.add("btn")
        btn_yes.style = "margin-left: -3px"
        parent.appendChild(btn_yes)
        const btn_idk = document.createElement("a")
        btn_idk.textContent = "???"
        btn_idk.classList.add("btn")
        btn_idk.style = "margin-left: 90px"

        parent.appendChild(btn_idk)
        btn_idk.addEventListener("click", (evt) => idk(data, indexes, count))

        const btn_no = document.createElement("a")
        btn_no.textContent = "NO "
        btn_no.classList.add("btn")
        btn_no.style = "margin-left: 90px"

        parent.appendChild(btn_no)
        btn_yes.addEventListener("click", (evt) => lymph(data, indexes, count))
        btn_no.addEventListener("click", (evt) => other(data, indexes, count))







    })
}

