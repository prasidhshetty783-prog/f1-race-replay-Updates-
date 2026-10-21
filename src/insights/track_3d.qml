// 3D Track Detail scene (Qt Quick 3D). Driven entirely from Python:
//   bridge                  - SceneBridge (camera mode, framing, status, labels)
//   groundGeometry, trackGeometry, edgeGeometry, drsGeometry, buildingGeometry
//                           - meshes built in src/lib/track_3d.py
//   carModel                - one row per car (position, yaw, team colour)
// Scene units are metres, Y up.

import QtQuick
import QtQuick3D
import QtQuick3D.Helpers

Item {
    id: root
    property int labelTick: 0

    // Dusk sky behind the transparent 3D view
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0a1220" }
            GradientStop { position: 0.6; color: "#1d2a3d" }
            GradientStop { position: 1.0; color: "#2b3a50" }
        }
    }

    View3D {
        id: view
        anchors.fill: parent
        camera: bridge.cameraMode === "drone" ? droneCamera
              : bridge.cameraMode === "follow" ? followCamera : orbitCamera

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Transparent
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            fog: Fog {
                enabled: true
                color: "#26344a"
                depthEnabled: true
                depthNear: bridge.radius * 1.4
                depthFar: bridge.radius * 4.5
            }
        }

        // ── Lights ───────────────────────────────────────────────────
        DirectionalLight {
            eulerRotation.x: -42
            eulerRotation.y: -32
            brightness: 1.15
            castsShadow: true
            shadowMapQuality: Light.ShadowMapQualityVeryHigh
            shadowFactor: 22
            shadowBias: 40
            shadowMapFar: bridge.radius * 4
        }
        DirectionalLight {           // soft fill so shaded sides aren't black
            eulerRotation.x: -65
            eulerRotation.y: 150
            brightness: 0.45
        }

        // ── Ground: terrain draped with the real-world map ───────────
        Model {
            geometry: groundGeometry
            receivesShadows: true
            materials: PrincipledMaterial {
                baseColor: bridge.hasGround ? "#ffffff" : "#18261d"
                baseColorMap: bridge.hasGround ? groundTexture : null
                roughness: 0.96
                specularAmount: 0.05
                cullMode: Material.NoCulling
            }
        }
        Texture {
            id: groundTexture
            source: bridge.groundSource
            generateMipmaps: true
            mipFilter: Texture.Linear
            tilingModeHorizontal: Texture.ClampToEdge
            tilingModeVertical: Texture.ClampToEdge
        }

        Model {                      // flat surroundings out to the horizon
            visible: bridge.ready
            source: "#Rectangle"
            position: Qt.vector3d(bridge.centre.x, bridge.outerLevel - 0.3, bridge.centre.z)
            eulerRotation.x: -90
            scale: Qt.vector3d(bridge.radius * 0.24, bridge.radius * 0.24, 1)
            materials: PrincipledMaterial { baseColor: "#121316"; roughness: 1.0; cullMode: Material.NoCulling }
        }

        // ── Track ────────────────────────────────────────────────────
        Model {
            geometry: trackGeometry
            receivesShadows: true
            materials: PrincipledMaterial {
                baseColor: "#33363d"
                roughness: 0.72
                cullMode: Material.NoCulling
            }
        }
        Model {
            geometry: edgeGeometry
            materials: PrincipledMaterial {
                baseColor: "#e6e6e6"
                roughness: 0.5
                cullMode: Material.NoCulling
            }
        }
        Model {
            geometry: drsGeometry
            materials: PrincipledMaterial {
                baseColor: "#1fae55"
                emissiveFactor: Qt.vector3d(0.0, 0.08, 0.03)
                roughness: 0.6
                cullMode: Material.NoCulling
            }
        }
        Node {                       // start / finish line across the track
            visible: bridge.ready
            position: bridge.startPos
            eulerRotation.y: bridge.startYaw
            Model {
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(bridge.trackHalfWidth * 2.1 / 100, 1.6 / 100, 1)
                materials: PrincipledMaterial { baseColor: "#ffffff"; lighting: PrincipledMaterial.NoLighting; cullMode: Material.NoCulling }
            }
        }

        // ── Buildings (OpenStreetMap footprints, extruded) ───────────
        Model {
            visible: bridge.hasBuildings
            geometry: buildingGeometry
            castsShadows: true
            receivesShadows: false
            materials: PrincipledMaterial {
                baseColor: "#6e7684"
                roughness: 0.88
                cullMode: Material.NoCulling
            }
        }

        // ── Cars ─────────────────────────────────────────────────────
        Repeater3D {
            model: carModel
            delegate: Car {
                position: Qt.vector3d(model.px, model.py, model.pz)
                eulerRotation.y: model.yaw
                bodyColour: model.colour
                highlighted: model.selected
                scale: Qt.vector3d(bridge.carScale, bridge.carScale, bridge.carScale)
            }
        }

        // ── Cameras ──────────────────────────────────────────────────
        Node {
            id: orbitOrigin
            objectName: "orbitOrigin"
            position: bridge.centre
        }
        PerspectiveCamera {
            id: orbitCamera
            objectName: "orbitCamera"
            parent: orbitOrigin
            position: Qt.vector3d(0, bridge.radius * 0.95, bridge.radius * 1.35)
            eulerRotation.x: -34
            fieldOfView: 45
            clipNear: 8
            clipFar: bridge.radius * 14
        }
        PerspectiveCamera {
            id: droneCamera
            objectName: "droneCamera"
            position: Qt.vector3d(bridge.centre.x - bridge.radius * 0.6, bridge.centre.y + 120,
                                  bridge.centre.z + bridge.radius * 0.6)
            eulerRotation.x: -18
            eulerRotation.y: -45
            fieldOfView: 60
            clipNear: 1
            clipFar: bridge.radius * 14
        }
        Node {
            id: followRig
            position: bridge.followPos
            eulerRotation.y: bridge.followYaw
            PerspectiveCamera {
                id: followCamera
                position: Qt.vector3d(0, 3.4, 11.5)
                eulerRotation.x: -8
                fieldOfView: 58
                clipNear: 0.3
                clipFar: bridge.radius * 14
            }
        }
    }

    OrbitCameraController {
        anchors.fill: parent
        origin: orbitOrigin
        camera: orbitCamera
        automaticClipping: false
        enabled: bridge.cameraMode === "orbit"
        visible: enabled
    }

    WasdController {
        id: wasd
        controlledObject: droneCamera
        speed: Math.max(0.5, bridge.radius / 600)
        shiftSpeed: 4
        enabled: bridge.cameraMode === "drone"
        keysEnabled: enabled
        mouseEnabled: enabled
    }

    // ── Corner numbers (2D labels that track their 3D position) ──────
    Timer { interval: 40; running: !!bridge && bridge.ready; repeat: true; onTriggered: root.labelTick++ }
    Repeater {
        model: bridge.corners
        delegate: Rectangle {
            property vector3d p: { root.labelTick; return view.mapFrom3DScene(Qt.vector3d(modelData.x, modelData.y, modelData.z)) }
            visible: !!bridge && bridge.ready && bridge.showCorners && p.z > 0
                     && (bridge.cameraMode !== "follow" || p.z < 600) && p.x > -20 && p.x < root.width + 20 && p.y > -20 && p.y < root.height + 20
            x: p.x - width / 2
            y: p.y - height - 6
            width: cornerText.implicitWidth + 10
            height: 18
            radius: 9
            color: "#cc0f1218"
            border.color: "#55ffffff"
            Text {
                id: cornerText
                anchors.centerIn: parent
                text: modelData.label
                color: "#f1f3f5"
                font.pixelSize: 11
                font.bold: true
            }
        }
    }

    // ── Camera mode switch ───────────────────────────────────────────
    Row {
        x: 14; y: 14
        spacing: 6
        Repeater {
            model: [["orbit", "Orbit"], ["follow", "Follow car"], ["drone", "Drone"]]
            delegate: Rectangle {
                width: label.implicitWidth + 26
                height: 30
                radius: 15
                color: bridge.cameraMode === modelData[0] ? "#e10600" : "#cc161a22"
                border.color: bridge.cameraMode === modelData[0] ? "#ff5a4f" : "#3a3f48"
                Text {
                    id: label
                    anchors.centerIn: parent
                    text: modelData[1]
                    color: "#ffffff"
                    font.pixelSize: 13
                    font.bold: true
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        bridge.setCameraMode(modelData[0])
                        if (modelData[0] === "drone")
                            wasd.forceActiveFocus()
                    }
                }
            }
        }
    }

    Text {
        x: 16; y: 52
        visible: bridge.cameraMode === "follow" && bridge.followCode !== ""
        text: "Following " + bridge.followCode
        color: "#e6e9ee"
        font.pixelSize: 13
        font.bold: true
        style: Text.Outline; styleColor: "#000000"
    }

    // ── Hints, status, credit ────────────────────────────────────────
    Rectangle {
        anchors.left: parent.left; anchors.bottom: parent.bottom
        anchors.margins: 12
        width: hint.implicitWidth + 20; height: 26; radius: 13
        color: "#b30d1015"
        Text {
            id: hint
            anchors.centerIn: parent
            color: "#c4c9d1"
            font.pixelSize: 12
            text: bridge.cameraMode === "drone"
                  ? "W A S D fly  ·  R / F up / down  ·  drag to look  ·  Shift = faster"
                  : bridge.cameraMode === "follow"
                    ? "Pick a car in the panel, or click one on the replay leaderboard"
                    : "Drag to rotate  ·  right-drag to pan  ·  scroll to zoom"
        }
    }

    Text {
        anchors.right: parent.right; anchors.bottom: parent.bottom
        anchors.margins: 10
        visible: bridge.mapCredit !== ""
        text: bridge.mapCredit
        color: "#a7adb6"
        font.pixelSize: 11
        style: Text.Outline; styleColor: "#000000"
    }

    Text {
        anchors.centerIn: parent
        visible: !bridge.ready
        text: bridge.status
        color: "#c4c9d1"
        font.pixelSize: 16
    }

    // ── Procedural F1 car (metres, nose towards -Z) ──────────────────
    component Car: Node {
        property color bodyColour: "#cccccc"
        property bool highlighted: false
        readonly property color carbon: "#1b1d22"

        Model {   // tub + engine cover
            source: "#Cube"
            position: Qt.vector3d(0, 0.42, 0.35)
            scale: Qt.vector3d(0.0085, 0.0048, 0.040)
            materials: PrincipledMaterial { baseColor: bodyColour; roughness: 0.32; metalness: 0.15 }
        }
        Model {   // sidepods
            source: "#Cube"
            position: Qt.vector3d(0, 0.34, 0.55)
            scale: Qt.vector3d(0.0165, 0.0042, 0.0175)
            materials: PrincipledMaterial { baseColor: bodyColour; roughness: 0.32; metalness: 0.15 }
        }
        Model {   // nose
            source: "#Cube"
            position: Qt.vector3d(0, 0.32, -2.05)
            scale: Qt.vector3d(0.0042, 0.0028, 0.0145)
            materials: PrincipledMaterial { baseColor: bodyColour; roughness: 0.32; metalness: 0.15 }
        }
        Model {   // front wing
            source: "#Cube"
            position: Qt.vector3d(0, 0.12, -2.72)
            scale: Qt.vector3d(0.019, 0.0016, 0.0042)
            materials: PrincipledMaterial { baseColor: carbon; roughness: 0.55 }
        }
        Model {   // rear wing
            source: "#Cube"
            position: Qt.vector3d(0, 0.92, 2.32)
            scale: Qt.vector3d(0.0105, 0.0022, 0.0045)
            materials: PrincipledMaterial { baseColor: carbon; roughness: 0.55 }
        }
        Model {   // halo / cockpit
            source: "#Cube"
            position: Qt.vector3d(0, 0.70, -0.25)
            scale: Qt.vector3d(0.0050, 0.0016, 0.0070)
            materials: PrincipledMaterial { baseColor: carbon; roughness: 0.4 }
        }
        Repeater3D {
            model: [Qt.vector3d(-0.86, 0.34, -1.62), Qt.vector3d(0.86, 0.34, -1.62),
                    Qt.vector3d(-0.84, 0.36, 1.58), Qt.vector3d(0.84, 0.36, 1.58)]
            delegate: Model {
                source: "#Cylinder"
                position: modelData
                eulerRotation.z: 90
                scale: Qt.vector3d(0.0068, 0.0040, 0.0068)
                materials: PrincipledMaterial { baseColor: "#121316"; roughness: 0.92 }
            }
        }
        Model {   // ground ring for the followed car
            visible: highlighted
            source: "#Cylinder"
            position: Qt.vector3d(0, 0.03, 0)
            scale: Qt.vector3d(0.05, 0.0004, 0.05)
            materials: PrincipledMaterial { baseColor: "#ffffff"; opacity: 0.35; lighting: PrincipledMaterial.NoLighting }
        }
    }
}
