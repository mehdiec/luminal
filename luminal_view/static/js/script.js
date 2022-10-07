document.getElementById("tabset").style.display = "none"

function create_image(img64) {
    const domImg = document.createElement("img")
    domImg.src = "data:image/png;base64," + img64

    return domImg
}
function imageOnClick(elementRef) {
    var posObject = getEventLocation(event);
}
function getEventLocation(eventRef) {
    var bcr = event.target.getBoundingClientRect()
    return { x: (eventRef.clientX - bcr.left) / (bcr.right - bcr.left), y: (eventRef.clientY - bcr.top) / (bcr.bottom - bcr.top) };
}
function clickOnGal(evt, element) {
    const parent = document.getElementById("gradcam_galery")
    parent.innerHTML = ""
    const image = document.createElement("img")
    image.src = "data:image/png;base64," + element.image
    // mettre autre id
    // document.getElementById("pred_galery").appendChild(getWidgetPrediction(element["prediction"]))
    const btn_gradcam = document.createElement("a")
    btn_gradcam.textContent = "GradCam"
    btn_gradcam.classList.add("btn")
    let position = document.createElement("h2")
    position.textContent = "position" + "(" + element.pos_x + " ," + element.pos_y + ")"
    console.log(element.pos_x, element.pos_y)

    parent.appendChild(position)

    // <a class="btn" onclick="run()">Run</a>
    const prediction = element["prediction"]
    parent.appendChild(getWidgetPrediction(prediction))
    parent.appendChild(image)
    parent.appendChild(btn_gradcam)
    btn_gradcam.addEventListener("click", (evt) => get_gradcam(element.pos_x, element.pos_y, document.getElementById("filename").value, (data) => {
        parent.innerHTML = ""
        let position = document.createElement("h2")
        position.textContent = "position" + "(" + element.pos_x + " ," + element.pos_y + ")"
        console.log(element.pos_x, element.pos_y)

        parent.appendChild(position)


        parent.appendChild(getWidgetPrediction(prediction))
        parent.appendChild(WidgetHeatmap("gradcamWidget", element.image, data))

    }))

}
function get_gradcam(x, y, file_name, cb) {
    const value = { "file_name": file_name, "x": x, "y": y, "model_name": "/data/DeepLearning/mehdi/top_gear/epoch=10-val_loss_ce=0.000.ckpt" }

    const param = new URLSearchParams(value)
    fetch("/gradcam?" + param).then((resp) => resp.json()).then(data => cb(data))
}
function run() {
    document.getElementById("main").style.display = "none"
    const value = { "file_name": document.getElementById("filename").value, "model_name": document.getElementById("model_name").value }
    const param = new URLSearchParams(value)
    fetch("/prediction?" + param).then((resp) => resp.json()).then(function (data) {
        document.getElementById("tabset").style.display = null
        var parent = document.getElementById("hist")
        var count = 0
        for (const label in data.hist) {
            console.log(data.hist[label])
            parent.appendChild(create_image(data.hist[label]))
            // document.getElementById("hist" + count).src = "data:image/png;base64," + data.hist[label]
            count++
        }

        // for (let i = 0; i < data.heatmap.length; i++) {
        //     parent.appendChild(create_image(data.heatmap[i]))
        // }
        parent = document.getElementById("patch0")
        for (const label in data.top) {
            let title = document.createElement("h2")
            title.textContent = label
            parent.appendChild(title)
            data.top[label].forEach(element => {
                let image = create_image(element.image)
                image.addEventListener("click", (evt) => clickOnGal(evt, element))
                parent.appendChild(image)
            });
        }
        document.getElementById("pie").src = "data:image/png;base64," + data.piechart["image"]
        document.getElementById("pie").style = "max-width:100%; height :auto"
        document.getElementById("pred_mean").appendChild(getWidgetPrediction(data.piechart["prediction"]))

        const baseSlide = data.heatmap["Raw Slide"]
        delete data.heatmap["Raw Slide"]
        document.getElementById("label_heat").appendChild(WidgetHeatmap("label_heat", baseSlide, data.heatmap, (x, y) => {
            const containerGradcam = document.getElementById("gradcam_custom")
            containerGradcam.innerHTML = ""
            get_gradcam(x, y, document.getElementById("filename").value, (dataGrad) => {
                containerGradcam.innerHTML = ""
                const basePatch = dataGrad["patch raw"]
                delete dataGrad["patch raw"]
                containerGradcam.appendChild(WidgetHeatmap("customgradcamWidget", basePatch, dataGrad))
            })
        }))
    })

}
function WidgetHeatmap(id, baseImg, DictHeatmap, clickEvt) {
    let container = document.createElement("div")
    container.style = "width:100%; display: flex; flex-direction: column;"
    let viewerContainer = document.createElement("div")
    viewerContainer.style.display = "flex"
    container.appendChild(viewerContainer)
    let heatmapViewer = document.createElement('div')
    heatmapViewer.classList.add("heatmapView")
    viewerContainer.appendChild(heatmapViewer)
    let imgHeatmap = document.createElement('img')
    imgHeatmap.classList.add("heatmap")
    heatmapViewer.appendChild(imgHeatmap)
    let imgBase = document.createElement('img')
    imgBase.src = "data:image/png;base64," + baseImg
    heatmapViewer.appendChild(imgBase)
    let sliderOpacity = document.createElement('input')
    sliderOpacity.type = "range"
    sliderOpacity.setAttribute('orient', "vertical")
    sliderOpacity.addEventListener('input', (evt) => { imgHeatmap.style.opacity = parseInt(sliderOpacity.value) / 100 * 0.7; })
    viewerContainer.appendChild(sliderOpacity)
    let containerHeatmapSelector = document.createElement('section')
    containerHeatmapSelector.classList.add("radioBtn")
    Object.keys(DictHeatmap).forEach((k, i) => {
        let btnContainer = document.createElement('div')
        let inputRadio = document.createElement('input')
        inputRadio.type = "radio"
        inputRadio.name = id + "_heatmapSelector"
        inputRadio.id = id + "_heatmapSelector_" + i
        inputRadio.value = k
        inputRadio.addEventListener('change', (evt) => { imgHeatmap.src = "data:image/png;base64," + DictHeatmap[k] })

        btnContainer.appendChild(inputRadio)
        let inputLabel = document.createElement('label')
        inputLabel.textContent = k
        inputLabel.setAttribute('for', id + "_heatmapSelector_" + i)
        btnContainer.appendChild(inputLabel)
        if (i == 0) {
            inputRadio.checked = true
            imgHeatmap.src = "data:image/png;base64," + DictHeatmap[k]
        }
        containerHeatmapSelector.appendChild(btnContainer)
    });
    container.appendChild(containerHeatmapSelector)
    if (clickEvt) {
        imgHeatmap.addEventListener("click", (evt) => {
            const coord = getEventLocation(evt)
            clickEvt(coord.x, coord.y)
        })
    }
    return container
}

function getWidgetPrediction(data) {
    var svgNS = "http://www.w3.org/2000/svg";
    drawRect = (x, y, w, h) => {
        const shape = document.createElementNS(svgNS, "rect");
        shape.setAttributeNS(null, "rx", 1);
        shape.setAttributeNS(null, "ry", 1);
        shape.setAttributeNS(null, "x", x);
        shape.setAttributeNS(null, "y", y);
        shape.setAttributeNS(null, "width", w);
        shape.setAttributeNS(null, "height", h);
        shape.setAttributeNS(null, "fill", 'none');
        return shape;
    }
    drawText = (x, y, size, text) => {
        const newText = document.createElementNS(svgNS, "text");
        newText.setAttributeNS(null, "x", x);
        newText.setAttributeNS(null, "y", y);
        newText.setAttributeNS(null, "font-size", size);
        newText.setAttributeNS(null, "font-family", "Arial, sans-serif")
        const textNode = document.createTextNode(text);
        newText.appendChild(textNode);
        return newText
    }
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    //svg.setAttribute('fill', 'none');
    svg.setAttribute('viewBox', '0 0 100 33');
    tabColor = ['#45B8DE', '#DEA845', '#888']
    Object.keys(data).forEach((k, i) => {
        const b = drawRect(0, (i + i * 0.2) * 4 + 0.2, 70, 4)
        b.setAttributeNS(null, "stroke", 'black');
        b.setAttributeNS(null, "stroke-width", 0.2)
        const p = drawRect(0, (i + i * 0.2) * 4 + 0.2, 70 * parseFloat(data[k]), 4)
        p.setAttributeNS(null, "fill", tabColor[i]);
        const txt = drawText(72, (i + i * 0.2) * 4 + 3.7, 2.5, (parseFloat(data[k]) * 100).toFixed(2) + '% ' + k)
        svg.appendChild(p)
        svg.appendChild(b)
        svg.appendChild(txt)
    })
    return svg
}


